import json
import os

from typing import Any, Dict, List
from ollama import Client

client = Client(
    host='https://ollama.com',
    headers={'Authorization': 'Bearer ' + os.environ.get('OLLAMA_API_KEY', '')}
)

MODEL = os.environ.get('OLLAMA_MODEL')

_SYSTEM_PROMPT = """You are a helpful SHL assessment advisor.
Your job is to produce a clear, concise final reply to the user based on:
  • The detected intent of the conversation
  • The output produced by the specialized sub-agent
  • Any additional RAG context retrieved from the catalog

Guidelines:
- Be conversational, friendly, and concise.
- If the intent is "clarify"   : ask exactly ONE focused follow-up question.
- If the intent is "compare"   : give a brief structured comparison (2-4 bullet points per assessment).
- If the intent is "recommend" : introduce the shortlist with 1-2 sentences; do NOT repeat the full list — the items are returned separately in the API response.
- If the intent is "refuse"    : politely explain the scope limitation; do NOT make up assessments.
- Never invent assessment names, URLs, or durations that are not in the provided context.
- Respond in plain text only; no markdown headers, no JSON.
"""

def generate_final_response(conversation: List[Dict[str, str]],intent: str,
    agent_output: Dict[str, Any],
) -> str:
    context_block = {
        "intent": intent,
        "agent_action": agent_output.get("action", intent),
        "agent_draft_reply": agent_output.get("reply", ""),
        "recommendations": [
            {"name": item.name, "test_type": item.test_type, "url": item.url}
            if hasattr(item, "name")
            else item
            for item in agent_output.get("items", [])
        ],
        "rag_context": agent_output.get("rag_context", {}),
    }

    user_prompt = (
        "Below is the full conversation history and the structured output from "
        "the sub-agent pipeline. Use this to write the final reply to the user.\n\n"
        f"=== PIPELINE OUTPUT ===\n{json.dumps(context_block, indent=2)}\n\n"
        "=== CONVERSATION HISTORY ===\n"
        + "\n".join(
            f"[{msg['role'].upper()}]: {msg['content']}"
            for msg in conversation
        )
        + "\n\nNow write the final reply to the user (plain text only):"
    )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]

    response = client.chat(
        model=MODEL,
        messages=messages,
        stream=False,
    )

    final_text: str = response.message.content.strip()
    return final_text or agent_output.get("reply", "")
