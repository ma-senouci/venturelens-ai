import uuid
from datetime import datetime

import pytest

from intake import build_run_input


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
        from models import RunInput
        assert isinstance(result, RunInput)
