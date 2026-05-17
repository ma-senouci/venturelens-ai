import importlib
import sys
from unittest.mock import MagicMock, patch

from models import (
    AgentFindings,
    CompetitionFindings,
    CriticFindings,
    MarketFindings,
    ProductFindings,
    RiskFindings,
    StageResult,
)


def _completed_stage(stage_name, key_findings=None, sources=None, evidence_gaps=None):
    findings_cls = {
        "market": MarketFindings,
        "competition": CompetitionFindings,
        "product": ProductFindings,
        "risk": RiskFindings,
    }.get(stage_name, AgentFindings)
    return StageResult(
        stage_name=stage_name,
        status="completed",
        findings=findings_cls(
            key_findings=key_findings if key_findings is not None else [f"{stage_name} finding 1"],
            sources=sources if sources is not None else [f"https://example.com/{stage_name}"],
            evidence_gaps=evidence_gaps if evidence_gaps is not None else [],
            confidence=0.8,
        ),
    )


def _failed_stage(stage_name, error="API rate limit exceeded"):
    return StageResult(stage_name=stage_name, status="failed", error=error)


def _completed_critic_stage(sources=None):
    return StageResult(
        stage_name="critic",
        status="completed",
        findings=CriticFindings(
            contradictions=["Conflicting retention signals"],
            weak_assumptions=["Expansion assumptions are thin"],
            unsupported_claims=["Category leadership is unproven"],
            open_questions=["What is net revenue retention?"],
            sources=sources if sources is not None else ["https://example.com/critic"],
            confidence=0.7,
            missing_perspectives=[],
        ),
    )


class TestRenderFindingsDisplay:
    def test_renders_one_section_per_completed_stage(self):
        stage_results = [
            _completed_stage("market"),
            _completed_stage("competition"),
        ]

        with patch("app.st") as mock_st:
            from app import render_findings_display

            render_findings_display(stage_results)

        expander_calls = mock_st.expander.call_args_list
        labels = [c[0][0] for c in expander_calls]
        assert any("Market Research" in label for label in labels)
        assert any("Competition Analysis" in label for label in labels)
        assert len(expander_calls) == 2

    def test_failed_stage_renders_caption_not_expander(self):
        stage_results = [
            _completed_stage("market"),
            _failed_stage("competition", error="Timeout"),
        ]

        with patch("app.st") as mock_st:
            from app import render_findings_display

            render_findings_display(stage_results)

        assert mock_st.expander.call_count == 1
        caption_calls = mock_st.caption.call_args_list
        caption_text = caption_calls[0][0][0]
        assert "Competition Analysis" in caption_text
        assert "Timeout" in caption_text

    def test_shows_key_findings_as_markdown(self):
        stage_results = [
            _completed_stage("market", key_findings=["Growing TAM", "Strong team"]),
        ]

        with patch("app.st") as mock_st:
            from app import render_findings_display

            render_findings_display(stage_results)

        md_calls = [c[0][0] for c in mock_st.markdown.call_args_list]
        rendered = "\n".join(md_calls)
        assert "Growing TAM" in rendered
        assert "Strong team" in rendered

    def test_shows_sources_once_per_section(self):
        stage_results = [
            _completed_stage(
                "market",
                key_findings=["Growing TAM", "Strong team"],
                sources=["https://crunchbase.com/acme", "https://techcrunch.com/acme"],
            ),
        ]

        with patch("app.st") as mock_st:
            from app import render_findings_display

            render_findings_display(stage_results)

        md_calls = [c[0][0] for c in mock_st.markdown.call_args_list]
        source_calls = [call for call in md_calls if "🔗" in call]
        assert len(source_calls) == 2
        assert "- 🔗 https://crunchbase.com/acme" in source_calls
        assert "- 🔗 https://techcrunch.com/acme" in source_calls
        assert "**Sources for this section:**" in md_calls

    def test_shows_evidence_gaps_in_warning(self):
        stage_results = [
            _completed_stage("market", evidence_gaps=["No public financials"]),
        ]

        with patch("app.st") as mock_st:
            from app import render_findings_display

            render_findings_display(stage_results)

        warning_calls = mock_st.warning.call_args_list
        assert len(warning_calls) == 1
        assert "No public financials" in warning_calls[0][0][0]

    def test_ignores_completed_critic_stage_in_research_findings_display(self):
        stage_results = [
            _completed_stage("market", key_findings=["Growing TAM"]),
            _completed_critic_stage(),
        ]

        with patch("app.st") as mock_st:
            from app import render_findings_display

            render_findings_display(stage_results)

        expander_calls = mock_st.expander.call_args_list
        labels = [call[0][0] for call in expander_calls]
        assert len(expander_calls) == 1
        assert any("Market Research" in label for label in labels)
        assert all("critic" not in label.casefold() for label in labels)


class TestRenderConsolidatedSources:
    def test_deduplicates_sources_across_stages(self):
        stage_results = [
            _completed_stage("market", sources=["https://a.com", "https://b.com"]),
            _completed_stage("competition", sources=["https://b.com", "https://c.com"]),
            _completed_stage("product", sources=["https://c.com", "https://d.com"]),
            _completed_stage("risk", sources=["https://d.com", "https://e.com"]),
        ]

        with patch("app.st") as mock_st:
            from app import render_consolidated_sources

            render_consolidated_sources(stage_results)

        md_calls = [c[0][0] for c in mock_st.markdown.call_args_list]
        rendered = "\n".join(md_calls)
        assert rendered.count("https://b.com") == 1
        assert "https://a.com" in rendered
        assert "https://c.com" in rendered
        assert "https://e.com" in rendered

    def test_renders_with_completed_critic_stage_and_ignores_critic_sources(self):
        stage_results = [
            _completed_stage("market", sources=["https://a.com"]),
            _completed_stage("competition", sources=["https://b.com"]),
            _completed_stage("product", sources=["https://c.com"]),
            _completed_stage("risk", sources=["https://d.com"]),
            _completed_critic_stage(sources=["https://critic.com"]),
        ]

        with patch("app.st") as mock_st:
            from app import render_consolidated_sources

            render_consolidated_sources(stage_results)

        mock_st.expander.assert_called_once()
        md_calls = [c[0][0] for c in mock_st.markdown.call_args_list]
        rendered = "\n".join(md_calls)
        assert "https://a.com" in rendered
        assert "https://d.com" in rendered
        assert "https://critic.com" not in rendered

    def test_skips_rendering_when_no_sources(self):
        stage_results = [
            _completed_stage("market", sources=[]),
            _completed_stage("competition", sources=[]),
            _completed_stage("product", sources=[]),
            _completed_stage("risk", sources=[]),
        ]

        with patch("app.st") as mock_st:
            from app import render_consolidated_sources

            render_consolidated_sources(stage_results)

        mock_st.expander.assert_not_called()

    def test_skips_partial_runs_even_when_sources_exist(self):
        stage_results = [
            _completed_stage("market", sources=["https://a.com"]),
            _completed_stage("competition", sources=["https://b.com"]),
            _completed_stage("product", sources=["https://c.com"]),
        ]

        with patch("app.st") as mock_st:
            from app import render_consolidated_sources

            render_consolidated_sources(stage_results)

        mock_st.expander.assert_not_called()


class TestFindingsDisplayWiring:
    def test_findings_not_rendered_while_pipeline_running(self):
        analysis_run = MagicMock()
        analysis_run.stage_results = [
            StageResult(
                stage_name="market",
                status="completed",
                findings=AgentFindings(key_findings=["f1"], sources=["s1"], evidence_gaps=[], confidence=0.8),
            )
        ]
        progress = {
            "stage_results": [analysis_run.stage_results[0]],
            "pipeline_result": None,
        }
        fake_st = MagicMock()
        fake_st.session_state = {"analysis_run": analysis_run, "pipeline_progress": progress}
        fake_st.form_submit_button.return_value = False

        sys.modules.pop("app", None)
        with patch.dict(sys.modules, {"streamlit": fake_st}), patch("config.get_settings", return_value=object()):
            importlib.import_module("app")

        subheader_calls = [call[0][0] for call in fake_st.subheader.call_args_list]
        assert "Research Findings" not in subheader_calls
