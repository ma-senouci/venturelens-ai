import uuid
from datetime import datetime

import pytest

from intake import build_run_input, create_analysis_run
from models import RunInput


class TestBuildRunInputValid:
    def test_valid_startup_name_creates_run_input(self):
        result = build_run_input(startup_name="Acme Corp")
        assert result.startup_name == "Acme Corp"
        assert uuid.UUID(result.id).version == 4
        assert datetime.fromisoformat(result.created_at).tzinfo is not None

    def test_all_fields_populated(self):
        result = build_run_input(
            startup_name="Acme Corp",
            website_url="https://acme.com",
            description="Rocket delivery startup",
            thesis="Large underserved market",
            analysis_focus="market size",
        )
        assert result.startup_name == "Acme Corp"
        assert result.website_url == "https://acme.com"
        assert result.description == "Rocket delivery startup"
        assert result.thesis == "Large underserved market"
        assert result.analysis_focus == "market size"

    def test_startup_name_is_stripped(self):
        result = build_run_input(startup_name="  Acme Corp  ")
        assert result.startup_name == "Acme Corp"


class TestBuildRunInputOptionalDefaults:
    def test_optional_fields_default_to_none(self):
        result = build_run_input(startup_name="Acme Corp")
        assert result.website_url is None
        assert result.description is None
        assert result.thesis is None
        assert result.analysis_focus is None

    def test_empty_strings_become_none(self):
        result = build_run_input(
            startup_name="Acme Corp",
            website_url="",
            description="",
            thesis="",
            analysis_focus="",
        )
        assert result.website_url is None
        assert result.description is None
        assert result.thesis is None
        assert result.analysis_focus is None

    def test_whitespace_only_optional_fields_become_none(self):
        result = build_run_input(
            startup_name="Acme Corp",
            website_url="   ",
            description="\t\n",
            thesis="  ",
            analysis_focus=" ",
        )
        assert result.website_url is None
        assert result.description is None
        assert result.thesis is None
        assert result.analysis_focus is None


class TestBuildRunInputValidation:
    def test_blank_name_raises_error(self):
        with pytest.raises(ValueError, match="Startup name is required"):
            build_run_input(startup_name="")

    def test_whitespace_only_name_raises_error(self):
        with pytest.raises(ValueError, match="Startup name is required"):
            build_run_input(startup_name="   ")

    def test_result_is_run_input_instance(self):
        result = build_run_input(startup_name="Acme Corp")
        assert isinstance(result, RunInput)


class TestCreateAnalysisRun:
    def test_status_is_running(self, sample_run_input):
        result = create_analysis_run(sample_run_input)
        assert result.status == "running"

    def test_stage_results_empty(self, sample_run_input):
        result = create_analysis_run(sample_run_input)
        assert result.stage_results == []

    def test_memo_is_none(self, sample_run_input):
        result = create_analysis_run(sample_run_input)
        assert result.memo is None

    def test_preserves_input_data(self, sample_run_input):
        result = create_analysis_run(sample_run_input)
        assert result.input == sample_run_input
        assert result.input.startup_name == sample_run_input.startup_name
        assert result.input.website_url == sample_run_input.website_url
        assert result.input.description == sample_run_input.description

    def test_has_valid_uuid4_id(self, sample_run_input):
        result = create_analysis_run(sample_run_input)
        assert uuid.UUID(result.id).version == 4

    def test_id_differs_from_run_input_id(self, sample_run_input):
        result = create_analysis_run(sample_run_input)
        assert result.id != sample_run_input.id

    def test_has_valid_iso8601_created_at(self, sample_run_input):
        result = create_analysis_run(sample_run_input)
        assert datetime.fromisoformat(result.created_at).tzinfo is not None

    def test_works_with_minimal_run_input(self):
        minimal = RunInput(startup_name="Minimal")
        result = create_analysis_run(minimal)
        assert result.status == "running"
        assert result.input.startup_name == "Minimal"
        assert result.input.website_url is None
