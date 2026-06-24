# VentureLens AI

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![CrewAI](https://img.shields.io/badge/CrewAI-agents-purple.svg)](https://docs.crewai.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-red.svg)](https://streamlit.io/)
[![Tests](https://github.com/ma-senouci/venturelens-ai/actions/workflows/test.yml/badge.svg)](https://github.com/ma-senouci/venturelens-ai/actions/workflows/test.yml)

**AI-powered startup due diligence that challenges its own conclusions before you read them.**

VentureLens orchestrates specialized research agents across market, competition, product positioning, and risk — then runs an Independent Review to pressure-test the findings before they reach the final recommendation.

The result is a source-backed investment memo with a clear **Invest**, **Watch**, or **Pass** verdict and a confidence score calibrated to what the evidence actually supports.

> Most AI research tools stop at summarization. VentureLens separates research from skepticism into explicit pipeline stages, so the recommendation has already been challenged before you see it.

🚀 **Live Demo:**

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Try_It-blue?style=for-the-badge)](https://venturelensai.streamlit.app)

Note: The live demo does not include my API keys. To use the app, provide your own `OPENAI_API_KEY` and `SERPER_API_KEY` through the sidebar.

---

## How It Works

```
┌─────────────┐
│   Intake    │  Startup name + optional context
└──────┬──────┘
       ▼
┌───────────────────────────────────────────────────┐
│                Research Agents                    │
│  ┌────────┐  ┌─────────┐  ┌─────────┐  ┌──────┐   │
│  │ Market │  │ Compet. │  │ Product │  │ Risk │   │
│  └────────┘  └─────────┘  └─────────┘  └──────┘   │
└────────────────────────┬──────────────────────────┘
                         ▼
              ┌─────────────────────┐
              │ Independent Review  │  Pressure-tests findings
              │   (Critic Agent)    │  before the final verdict
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │   Memo Synthesis    │  Synthesizes everything
              │    (Memo Agent)     │  into the final memo
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │    Memo Output      │  Structured, source-backed,
              │                     │  ready for review
              └─────────────────────┘
```

### Key Design Decisions

- **Structured skepticism** — A dedicated critic agent pressure-tests all research outputs before the final recommendation, so weak reasoning is caught early.
- **Graceful degradation** — If an agent fails, the pipeline continues. Partial runs lower confidence explicitly rather than hiding gaps.
- **Source traceability** — Every major claim links back to cited sources. The memo distinguishes what was found from what remains unknown.
- **Evidence guardrails** — Findings without adequate sourcing are automatically demoted to evidence gaps rather than surfaced as conclusions.
- **Visible reasoning** — Real-time progress shows each stage as it executes. No black box.

---

## What the Memo Includes

Each completed run produces a structured memo containing:

- **Recommendation** — Invest, Watch, or Pass
- **Confidence score** — 0.0–1.0 reflecting evidence strength
- **Confidence factors** — what raised or lowered the score and why
- **Executive summary** — LLM-synthesized from all findings and critique
- **Research findings** — organized by market, competition, product positioning, and risk
- **Independent Review** — contradictions, weak assumptions, and unsupported claims
- **Unresolved risks** — what could go wrong that the evidence couldn't rule out
- **Open questions** — what the research couldn't answer
- **Consolidated sources** — cited references across all research stages

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Language** | Python 3.12 |
| **Package Manager** | [uv](https://docs.astral.sh/uv/) |
| **Agent Framework** | [CrewAI](https://docs.crewai.com/) |
| **Web Interface** | [Streamlit](https://streamlit.io/) |
| **Data Models** | [Pydantic](https://docs.pydantic.dev/) |
| **Persistence** | SQLite |
| **Search** | [Serper API](https://serper.dev/) (via crewai-tools) |
| **LLM** | OpenAI (configurable model) |
| **Testing** | pytest |
| **Linting** | ruff |

---

## Getting Started

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager
- An [OpenAI API key](https://platform.openai.com/api-keys)
- A [Serper API key](https://serper.dev/) (for web search)

### Quick Start

```bash
git clone https://github.com/ma-senouci/venturelens-ai.git
cd venturelens-ai
uv sync
cp .env.example .env   # Windows: copy .env.example .env
```

Edit `.env` with your API keys:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL_NAME=gpt-4o-mini        # optional, defaults to gpt-4o-mini
SERPER_API_KEY=your_serper_api_key_here
```

### Run

```bash
uv run streamlit run src/app.py
```

The app opens at `http://localhost:8501`. Enter a startup name, optionally add context (website, thesis, focus area), and hit **Prepare analysis** to review the inputs — then **Run analysis** to start the pipeline.

A typical run completes in **2–3 minutes**.

### Run Tests

```bash
uv run pytest
```

All tests run without API keys — external calls are mocked.

---

## ⚙️ Configuration

You can configure API keys through the sidebar UI or with a `.env` file. Values entered in the sidebar override matching values loaded from `.env`, so the `.env` file is optional.

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | Required for model inference. Can be provided via sidebar or `.env`. |
| `SERPER_API_KEY` | Yes | Required for web search. Can be provided via sidebar or `.env`. |
| `OPENAI_MODEL_NAME` | No | Override the default LLM model. Defaults to `gpt-4o-mini`. |

---

## Project Structure

```
venturelens-ai/
├── .github/
│   └── workflows/
│       └── test.yml              # CI — runs pytest on push/PR
├── src/
│   ├── app.py               # Streamlit UI — intake, progress, memo display
│   ├── orchestrator.py       # Pipeline orchestration — runs agents, manages callbacks
│   ├── pipeline_runner.py    # Session-aware pipeline entry point
│   ├── agents.py             # CrewAI agent definitions — market, competition, product, risk, critic
│   ├── memo_generator.py     # Hybrid memo synthesis — LLM narrative + programmatic fields
│   ├── models.py             # Pydantic domain models — data contracts for the pipeline
│   ├── persistence.py        # SQLite save/load/list operations
│   ├── config.py             # Settings from environment variables
│   └── intake.py             # Input validation and normalization
├── tests/                    # 15 test modules, ~100% critical-path coverage
├── data/                     # SQLite database (gitignored)
├── .env.example              # Required environment variables template
├── pyproject.toml            # Project config, dependencies, ruff & pytest settings
├── requirements.txt          # pip dependencies for Streamlit Cloud deployment
├── README.md                 # This file
└── LICENSE                   # MIT
```

---

## Architecture

Dependencies flow in one direction. `app.py` is the root — nothing imports from it. `models.py` and `config.py` are leaf nodes — they import nothing from the project.

```
app.py
├── intake.py
├── orchestrator.py
├── pipeline_runner.py
│   ├── agents.py
│   ├── memo_generator.py
│   └── orchestrator.py
└── persistence.py

Leaf nodes (import nothing from the project):
  models.py  ← imported by all modules above
  config.py  ← imported by app, pipeline_runner, agents, memo_generator
```

| Module | Responsibility |
|--------|---------------|
| `app.py` | UI rendering, session state, user interaction |
| `intake.py` | Input validation, normalization, and run creation |
| `pipeline_runner.py` | Wires agents, callbacks, and orchestrator for a single run |
| `orchestrator.py` | Executes pipeline stages, per-agent error handling, status resolution |
| `agents.py` | CrewAI agent/task definitions with structured Pydantic outputs |
| `memo_generator.py` | Builds the final memo — LLM for narrative, Python for everything else |
| `models.py` | All Pydantic data contracts shared across modules |
| `persistence.py` | SQLite CRUD for completed runs |
| `config.py` | Environment variable loading via pydantic-settings |

---

## Testing Coverage

The repository includes 15 test modules with 153 test cases covering:

- agent wrappers, tool binding, and error handling
- critic agent review and structured output validation
- orchestrator success, failure, partial-completion, and callback paths
- pipeline runner thread lifecycle and settings wiring
- memo generator — LLM narrative synthesis + programmatic confidence scoring
- persistence save, load, list, and upsert operations
- config validation, missing keys, blank values, and model name defaults
- API key sidebar/environment resolution and override behavior
- app-level intake form, progress display, and run history rendering
- memo display — confidence warnings, partial-run handling, source rendering
- findings display — expandable sections, failed stages, evidence gaps
- Pydantic model validation — UUIDs, ISO timestamps, confidence bounds

All tests run without API keys — external calls are mocked.

---

## Why This Project

This project demonstrates agent engineering patterns that go beyond a single prompt-and-response demo:

- **Multi-agent pipeline** — specialized agents with typed handoffs via Pydantic models
- **Orchestration in Python** — pipeline logic lives outside the LLM, making it testable and debuggable
- **Built-in skepticism** — a dedicated critic agent reviews all research before the final verdict
- **Graceful degradation** — partial failures lower confidence explicitly rather than hiding gaps
- **Hybrid synthesis** — the memo combines LLM-generated narrative with programmatic confidence scoring
- **Real-time progress** — streaming UI shows each stage as it executes
- **Fully tested** — 15 test modules with mocked external calls, no API keys required

---

## License

[MIT](LICENSE) — Mohamed Abdelkrim Senouci
