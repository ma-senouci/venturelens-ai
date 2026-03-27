# VentureLens AI

AI-powered startup due diligence application that generates structured investment memos.

## What It Does

Submit a startup name and get a structured due diligence memo.

## Tech Stack

- Python 3.12 + uv
- CrewAI for multi-agent orchestration
- Streamlit for the web interface

## Getting Started

```bash
git clone https://github.com/ma-senouci/venturelens-ai.git
cd venturelens-ai
uv sync
cp .env.example .env
# Edit .env with your API keys
uv run streamlit run src/app.py
```

## Status

🚧 Under development

## License

MIT
