# 🧠 Persona Market

**Clone your expertise into an AI persona and sell access — like Cameo for knowledge.**

A doctor creates their AI clone trained on their medical knowledge → patients pay $5/consultation.  
A senior dev clones their debugging skills → junior developers pay to use it.  
Platform takes 20% commission. Creators keep 80%.

## Installation

```bash
git clone <repo-url> && cd persona
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env   # add your API keys
```

## CLI Usage

```bash
# Create a persona interactively
persona create

# Add knowledge from a file
persona train "Dr. Sarah" --file knowledge.yaml

# Chat with a persona
persona chat "Dr. Sarah"

# List all personas
persona list

# View earnings and stats
persona stats "Dr. Sarah"
```

## API Server

```bash
uvicorn persona.api.app:app --reload --port 8000
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/api/v1/personas` | Create persona |
| `GET` | `/api/v1/personas` | List / search personas |
| `GET` | `/api/v1/personas/{id}` | Get persona details |
| `POST` | `/api/v1/personas/{id}/chat` | Chat with persona |
| `POST` | `/api/v1/personas/{id}/train` | Add training data |
| `GET` | `/api/v1/personas/{id}/stats` | Usage statistics |

## MCP Integration

The project includes an MCP server for use with Claude Desktop or Cursor:

```json
{
  "mcpServers": {
    "persona-market": {
      "command": "python",
      "args": ["-m", "persona.mcp.server"]
    }
  }
}
```

### MCP Tools

- `create_persona` — Create a new AI persona
- `chat_with_persona` — Talk to a persona
- `list_personas` — Browse the marketplace
- `train_persona` — Add knowledge

## Training Data Format (YAML)

```yaml
text_samples:
  - "Paragraph of expert knowledge..."

qa_pairs:
  - question: "What is X?"
    answer: "X is a concept that..."

expertise_docs:
  - "Detailed document content..."

style_examples:
  - "Example of how the expert communicates..."
```

## Architecture

```
persona/
├── models.py        # Pydantic domain models
├── knowledge.py     # SQLite knowledge base
├── engine.py        # Core persona engine
├── marketplace.py   # Marketplace logic
├── ai.py            # LLM abstraction layer
├── cli.py           # Typer CLI
├── api/app.py       # FastAPI REST API
└── mcp/server.py    # MCP server
```

## Running Tests

```bash
pytest tests/ -v
```

## License

MIT
