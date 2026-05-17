import os

from typing import Any, Dict, List
from ollama import Client

client = Client(
    host='https://ollama.com',
    headers={'Authorization': 'Bearer ' + os.environ.get('OLLAMA_API_KEY', '')}
)

MODEL = os.environ.get("OLLAMA_MODEL")

SYSTEM_PROMPT_CLARIFY = """
You are an SHL assessment clarification assistant.

Your task is to ask ONE precise clarification question that helps narrow down SHL assessment recommendations.

You must identify the SINGLE most important missing piece of information from the conversation.

Possible missing information includes:
- seniority level
- hiring purpose (selection vs development)
- personality vs cognitive vs technical evaluation
- leadership benchmarking needs
- stakeholder interaction requirements
- job family or role type

Rules:
- Ask ONLY ONE question.
- Ask the MOST informative missing question.
- Keep the question under 20 words.
- Do NOT ask compound or multi-part questions.
- Do NOT recommend assessments.
- Do NOT mention products, bundles, or reports.
- Avoid generic questions like:
  "Tell me more about the role."
- Be direct and recruiter-focused.

GOOD QUESTIONS:
- Is this for selection or leadership development?
- What seniority level are these candidates?
- Do you want personality or cognitive assessment?
- Are you benchmarking candidates against leadership standards?

BAD QUESTIONS:
- What competencies and assessment types are important?
- Can you tell me more about your hiring needs?
- Which assessment package would you prefer?
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
