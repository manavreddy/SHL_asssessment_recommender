import os

from typing import Any, Dict, List
from ollama import Client

client = Client(
    host='https://ollama.com',
    headers={'Authorization': 'Bearer ' + os.environ.get('OLLAMA_API_KEY', '')}
)

MODEL = os.environ.get("OLLAMA_MODEL")

SYSTEM_PROMPT_CLARIFY = """You are a helpful assistant helping users find the right SHL assessments.

The user's request is unclear. Ask exactly ONE short, focused question to understand what is missing — for example:
- What role or seniority level they are hiring for
- What skills or traits they want to assess (cognitive ability, personality, situational judgment, etc.)
- Whether they need a specific test type or have time/format constraints

Rules:
- Ask only ONE question.
- Keep it under 30 words.
- Do NOT recommend any assessments yet.
"""

Conversation = List[Dict[str, str]]


def clarification_agent(conversation: Conversation) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_CLARIFY},
        *conversation,
    ]

    response = client.chat(
        model=MODEL,
        messages=messages,
        stream=False,
    )

    question = response.message.content.strip()

    return {
        "action": "clarify",
        "reply": question,
        "recommendations": [],
        "end_of_conversation": False,
    }
