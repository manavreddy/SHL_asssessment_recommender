import json
import os
import re
from typing import Any, Dict, List

from ollama import Client
from tools.get_assessment import get_assessment


client = Client(
    host="https://ollama.com",
    headers={"Authorization": "Bearer " + os.environ.get("OLLAMA_API_KEY", "")},
)

MODEL = os.environ.get("OLLAMA_MODEL")


REPEAT_EXTRACTION_PROMPT = """
You are an SHL assessment extraction assistant.

The previous assistant message contains assessment recommendations.
Extract only the SHL assessments that were recommended.

Return ONLY valid JSON with this schema:
{{
  "assessments": [
    {{
      "name": "assessment name",
      "url": "catalog URL if present, otherwise empty string",
      "test_type": "test type if present, otherwise empty string",
      "duration": "duration if present, otherwise empty string"
    }}
  ]
}}

Rules:
- Do not invent assessment names.
- If a URL is shown, copy it exactly.
- If a field is not present, use an empty string.
- Do not explain your reasoning.

Previous assistant message:
{assistant_message}
"""


def repeat_agent(conversation: List[Dict[str, str]]) -> Dict[str, Any]:
    previous_reply = _last_assistant_message(conversation)

    if not previous_reply.strip():
        return {
            "action": "clarify",
            "reply": "I do not see a previous shortlist to repeat.",
            "items": [],
            "end_of_conversation": False,
        }

    extracted_items = _extract_previous_recommendations(previous_reply)
    if not extracted_items:
        print("Repeat agent: LLM could not extract previous recommendations.")
        return {
            "action": "clarify",
            "reply": "I could not identify the previous recommendations to repeat.",
            "items": [],
            "end_of_conversation": False,
        }

    items = [_fill_from_catalog(item) for item in extracted_items]
    items = _dedupe_by_url_or_name(items)
    items = items[:5]

    return {
        "action": "repeat",
        "reply": "Great, locking in the same shortlist.",
        "items": items,
        "end_of_conversation": True,
    }


def _extract_previous_recommendations(previous_reply: str) -> List[Dict[str, Any]]:
    if not MODEL:
        print("Repeat agent: OLLAMA_MODEL is not set.")
        return []

    prompt = REPEAT_EXTRACTION_PROMPT.format(assistant_message=previous_reply)

    try:
        response = client.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        content = response.message.content.strip()
        data = json.loads(_json_object(content))
        return data.get("assessments", [])
    except Exception as error:
        print(f"Repeat extraction error: {error}")
        return []


def _fill_from_catalog(item: Dict[str, Any]) -> Dict[str, Any]:
    name = item.get("name", "")
    catalog_item = get_assessment(name) if name else None

    if not catalog_item:
        return _normalize_item(item)

    return {
        "name": catalog_item.get("name", item.get("name", "")),
        "url": catalog_item.get("url", item.get("url", "")),
        "test_type": catalog_item.get("test_type", item.get("test_type", "")),
        "duration": catalog_item.get("duration", item.get("duration", "")),
        "description": catalog_item.get("description", item.get("description", "")),
        "keys": catalog_item.get("keys", []),
        "languages": catalog_item.get("languages", []),
    }


def _normalize_item(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": item.get("name", ""),
        "url": item.get("url", ""),
        "test_type": item.get("test_type", ""),
        "duration": item.get("duration", ""),
        "description": item.get("description", ""),
        "keys": item.get("keys", []),
        "languages": item.get("languages", []),
    }


def _last_assistant_message(conversation: List[Dict[str, str]]) -> str:
    for message in reversed(conversation):
        if message.get("role") == "assistant":
            return message.get("content", "")
    return ""


def _dedupe_by_url_or_name(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique = []

    for item in items:
        key = item.get("url") or item.get("name", "").lower()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(item)

    return unique


def _json_object(text: str) -> str:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return match.group(0) if match else text
