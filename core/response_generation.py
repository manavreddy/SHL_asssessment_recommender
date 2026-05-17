import json
import os

from typing import Any, Dict, List
from ollama import Client

client = Client(
    host='https://ollama.com',
    headers={'Authorization': 'Bearer ' + os.environ.get('OLLAMA_API_KEY', '')}
)

MODEL = os.environ.get('OLLAMA_MODEL')

_SYSTEM_PROMPT = """
You are a grounded SHL assessment response generator.

Your task is to generate the final user-facing reply STRICTLY using:
- the conversation history
- the detected intent
- the structured sub-agent output
- the retrieved catalog-backed assessments

CRITICAL RULES:
- NEVER invent assessment names.
- NEVER invent bundles, packages, or suites.
- NEVER mention products not present in recommendations.
- NEVER use external SHL knowledge.
- NEVER synthesize new recommendations.
- Use ONLY the provided recommendation items.
- If recommendations are empty, ask ONE concise clarification question.
- Keep replies concise and recruiter-focused.
- Do NOT repeat the full recommendation list.
- Do NOT explain internal reasoning.
- Do NOT generate markdown or JSON.

Intent Rules:

1. clarify
- Ask exactly ONE short clarification question.
- Do not recommend assessments.

2. recommend
- Briefly explain why the retrieved assessments fit the user's needs.
- Mention only themes already present in retrieved items.
- Do not mention assessments absent from recommendations.

3. compare
- Give a short grounded comparison using ONLY provided catalog data.

4. refine
- Briefly explain that recommendations were refined based on updated requirements.

5. repeat
- Briefly confirm the shortlist again without adding new information.

6. refuse
- Politely explain limitations without inventing alternatives.

If unsure:
stay conservative and use only explicitly provided information.
"""

def generate_final_response(conversation: List[Dict[str, str]],intent: str,
    agent_output: Dict[str, Any],
) -> str:
    context_block = {
        "intent": intent,
        "agent_action": agent_output.get("action", intent),
        "agent_draft_reply": agent_output.get("reply", ""),
        "recommendations": [
            {"name": item.name, "test_type": item.test_type,
             "keys":item.keys, "languages":item.languages, 
             "description":item.description, "url": item.url}
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
