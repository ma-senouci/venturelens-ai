from models import AnalysisRun, RunInput


def build_run_input(
    startup_name: str,
    website_url: str = "",
    description: str = "",
    thesis: str = "",
    analysis_focus: str = "",
) -> RunInput:
    name = startup_name.strip()
    if not name:
        raise ValueError("Startup name is required")
    return RunInput(
        startup_name=name,
        website_url=website_url.strip() or None,
        description=description.strip() or None,
        thesis=thesis.strip() or None,
        analysis_focus=analysis_focus.strip() or None,
    )


def create_analysis_run(run_input: RunInput) -> AnalysisRun:
    return AnalysisRun(status="running", input=run_input, stage_results=[])
