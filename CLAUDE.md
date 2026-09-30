# Persona Market

Clone your expertise into an AI persona and sell access — like Cameo for knowledge.

## Quick Start

```bash
# Install
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'

# Configure
cp .env.example .env   # add your API keys

# Use CLI
persona create          # create a persona
persona list            # list personas
persona chat <name>     # chat with a persona
persona train <name> -f data.yaml   # add knowledge
persona stats <name>    # view earnings

# Run API server
uvicorn persona.api.app:app --reload

# Run tests
pytest tests/ -v
```

## Architecture

- `persona/models.py` — Pydantic domain models
- `persona/knowledge.py` — SQLite-backed knowledge base
- `persona/engine.py` — Persona creation, chat, training, evaluation
- `persona/marketplace.py` — Marketplace browsing and earnings
- `persona/ai.py` — LLM abstraction (Anthropic / OpenAI)
- `persona/cli.py` — Typer CLI with Rich output
- `persona/api/app.py` — FastAPI REST endpoints
- `persona/mcp/server.py` — MCP server for tool integration

## Economics

- Creators set a price per query (default $5.00)
- Platform takes 20% commission
- Creator receives 80% of revenue
