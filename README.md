# SHL Assessment Recommender

Simple v1 FastAPI service for the SHL AI Intern assignment.

## Setup

```bash
pip install -r requirements.txt
python build_index.py
uvicorn app.main:app --reload
```

## Endpoints

```text
GET /health
POST /chat
```

Example request:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "I am hiring a senior Java engineer with Spring and SQL experience."
    }
  ]
}
```

The response always follows the assignment schema:

```json
{
  "reply": "Here is a catalog-backed SHL shortlist...",
  "recommendations": [
    {
      "name": "Core Java (Advanced Level) (New)",
      "url": "https://www.shl.com/products/product-catalog/view/core-java-advanced-level-new/",
      "test_type": "K"
    }
  ],
  "end_of_conversation": false
}
```

## Agent Flow

```text
POST /chat
  -> orchestrate_chat() orchestrator
  -> build_state() intent + context analyzer
  -> select_agent() in orchestrator
  -> specialized_agents/* or refusal handler
  -> retrieval tools when recommendations are needed
  -> build_response() schema-compliant JSON
```

The orchestrator owns all routing. `specialized_agents/` contains the worker functions. Each specialized agent can call Ollama for wording, while catalog item selection stays deterministic. Pydantic classes are only used for API schemas.

## Ollama

Set these in `.env`:

```text
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1
OLLAMA_API_KEY=
```

`OLLAMA_API_KEY` is optional for local Ollama. If Ollama is unavailable, the app falls back to deterministic replies.
